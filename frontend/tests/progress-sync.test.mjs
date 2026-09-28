import test from "node:test";
import assert from "node:assert/strict";
import { syncedPercent, createDurationMemory, trackWork, IMMEDIATE_MS } from "../src/progress-sync.js";

test("percent follows elapsed time against the expected duration and never reaches 100 early", () => {
    assert.equal(syncedPercent(0, 10000), 0);
    assert.ok(Math.abs(syncedPercent(10000, 10000) - 82) <= 1); // ~86% of the 95 ceiling
    assert.ok(syncedPercent(30000, 10000) <= 95);
    // a slower stage crawls: the same elapsed time reads lower
    assert.ok(syncedPercent(5000, 30000) < syncedPercent(5000, 10000));
});

test("work that finishes immediately shows no readout", async () => {
    let shown = false;
    let ticks = 0;
    const r = await trackWork({ work: Promise.resolve(7), steps: ["a"], expectedMs: 1000, onShow: () => { shown = true; }, onTick: () => { ticks += 1; } });
    assert.equal(r.result, 7);
    assert.equal(r.shown, false);
    assert.equal(shown, false);
    assert.equal(ticks, 0);
});

test("slow work shows a readout that lasts as long as the work and ends at 100", async () => {
    const seen = [];
    const t0 = Date.now();
    const r = await trackWork({
        work: new Promise(resolve => setTimeout(() => resolve("ok"), IMMEDIATE_MS + 400)),
        steps: ["one", "two"],
        expectedMs: 5000,
        onTick: p => seen.push(p),
        tickMs: 50,
    });
    assert.equal(r.shown, true);
    assert.equal(r.result, "ok");
    assert.ok(Date.now() - t0 >= IMMEDIATE_MS + 400);
    assert.equal(seen.at(-1), 100);
    assert.ok(seen.every((p, i) => i === 0 || p >= seen[i - 1]), "monotonic");
});

test("a rejected work still settles", async () => {
    const r = await trackWork({ work: Promise.reject(new Error("x")), steps: [], expectedMs: 100, onTick: () => {} });
    assert.equal(r.result, undefined);
});

test("durations are remembered as a moving average, with defaults", () => {
    const store = new Map();
    const storage = { getItem: k => store.get(k) ?? null, setItem: (k, v) => store.set(k, v) };
    const m = createDurationMemory({ depth: 12000 }, storage);
    assert.equal(m.expected("depth"), 12000);
    m.record("depth", 20000);
    assert.equal(m.expected("depth"), 20000);
    m.record("depth", 10000);
    assert.equal(m.expected("depth"), 16000);
    assert.equal(createDurationMemory({}, storage).expected("depth"), 16000);
});
